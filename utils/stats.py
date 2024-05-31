def stats(epoch, step, loss_D1, loss_D2, loss_G):
    return f"Epoch [{epoch}], Step [{step}], D1 Loss: {loss_D1:.4f}, D2 Loss: {loss_D2:.4f}, G Loss: {loss_G:.4f}"